"""
Evaluate RoboVLMs-CoT via LIBERO Benchmark
"""

import traceback
import argparse
import json
import logging
from pathlib import Path
import time
from robovlms.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import sys
import os
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from pytorch_lightning import seed_everything
import torch
import torch.distributed as dist
from robovlms.utils.config_utils import load_config

from model_wrapper_emu import EmuVLAModel
from libero_utils import save_rollout_gif, get_libero_image, get_episode_length, get_libero_wrist_image, quat2axisangle
from libero_utils import get_libero_dummy_action, get_libero_env
from libero.libero import benchmark

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def setup():
    dist.init_process_group(backend="nccl")
    os.environ["EGL_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def evaluate(
    model,
    model_name=None,
    debug=False,
    resize_size=256,
    num_trials_per_task=50,
    num_steps_wait=10,
    local_log_dir=None,
    task_suite_name="libero_object",
):
    # Initialize Local Logging
    run_id = f"{task_suite_name}-{time.strftime('%Y-%m-%d_%H:%M')}"
    os.makedirs(local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"Task suite: {task_suite_name}")
    log_file.write(f"Task suite: {task_suite_name}\n")
    EP_LEN = get_episode_length(task_suite_name)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in range(num_tasks_in_suite):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)
        task_episodes, task_successes = 0, 0
        logger.info(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        for episode_idx in range(num_trials_per_task):
            env.reset()
            model.reset()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            if model.use_cot:
                thought = [""]

            # Start episodes
            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            action_counter = 0
            while t < EP_LEN + num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Prepare observation
                    observation, img = prepare_observation(obs, resize_size)
                    replay_images.append(img)
                    
                    if model.use_cot:
                        # Create a white background for the text
                        text_img = (
                            np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
                        )
                        # Split thought into multiple lines
                        lines = thought[0].replace("@", "\n").split("\n")
                        # Add text lines
                        for i, line in enumerate(lines):
                            cv2.putText(
                                text_img,
                                line,
                                (10, 30 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 0),
                                1,
                            )
                        # Concatenate original image with text image
                        img = np.concatenate((img, text_img), axis=1)
                        # Save a sample image for debugging
                        # cv2.imwrite("sample_cot_image.png", img)

                    if action_counter == 0:
                        if model.use_cot:
                            action, thought = model.step(obs_img, task_description)
                        else:
                            action = model.step(observation, task_description)
                            # from PIL import Image
                            # Image.fromarray(img).save(f"img_{t}_{action_counter}.png")
                            # Image.fromarray(observation['wrist_image']).save(f"wrist_{t}_{action_counter}.png")

                        action_counter = action.shape[0]

                    
                    # logger.info(f"Action: {action.shape}")
                    step_action = action[-action_counter]
                    obs, reward, done, info = env.step(step_action.tolist())
                    action_counter -= 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            logger.info(f"Num of Steps: {len(replay_images)}")
            if debug and len(replay_images) > 0:
                gif_dir = os.path.join(local_log_dir, "videos-{}".format(run_id))
                if not os.path.exists(gif_dir):
                    os.makedirs(gif_dir, exist_ok=True)
                gif_path = f"Episodes{total_episodes}_{str(done)}.gif"
                gif_path = os.path.join(gif_dir, gif_path)
                save_rollout_gif(replay_images, gif_path, fps=15)

            # Log current results
            logger.info(f"Success: {done}")
            logger.info(f"# episodes completed so far: {total_episodes}")
            logger.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        logger.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logger.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    log_file.close()

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Prepare observations dict
    observation = {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img 

def parser_args():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info and visualize environment.",
    )

    # yaml_path takes the highest priority, then the log_dir, finally the config_path
    parser.add_argument(
        "--config_path", type=str, default=None, help="path to the config file"
    )
    parser.add_argument(
        "--is_pt_config",
        action="store_true",
        help="whether the specified config path is a pretrain config file.",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        nargs="+",
        default="",
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="checkpoint directory of the training",
    )
    parser.add_argument(
        "--ckpt_idx", type=int, default=-1, help="which ckpt is going to be evaluated"
    )
    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument("--vq_hub", type=str, default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-Base")
    parser.add_argument("--vision_hub", type=str, default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-VisionVQ")
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="select evaluate LIBREO TASK SUITE",
    )
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to the normalization statistics JSON file. If not provided, will use default path relative to project root.")
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--no_nccl", action="store_true")
    parser.add_argument("--no_action_ensemble", action="store_true")
    parser.add_argument('--cache_root', type=str, default="/share/project/yuqi.wang/UniVLA/logs/libero",
                        help="Root directory to store cache/logs.")

    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    if not args.no_nccl:
        setup()

    CACHE_ROOT = args.cache_root
    os.makedirs(CACHE_ROOT, exist_ok=True)

    eval_log_dir = os.path.join(CACHE_ROOT, 'eval')
    os.makedirs(eval_log_dir, exist_ok=True)

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    model = EmuVLAModel(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub,
            device=torch.device("cuda"),
            norm_stats_path=args.norm_stats_path,
        )

    sr_path = os.path.join(eval_log_dir, f"success_rate_calvin.txt")
    result_path = os.path.join(
        eval_log_dir, f"results_calvin_rand-{args.rank}.json"
    )
    cache_file = os.path.join(eval_log_dir, f"meta_info.json")

    if not args.no_cache and args.local_rank == 0:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        with open(cache_file, "w") as f:
            _info = {
                "eval_sr_path": sr_path,
                "eval_result_path": result_path,
                "eval_log_dir": eval_log_dir,
            }
            json.dump(_info, f, indent=2)

    evaluate(
        model,
        task_suite_name=args.task_suite_name,
        local_log_dir=eval_log_dir,
        debug=args.debug,
    )

    if not args.no_nccl:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()
