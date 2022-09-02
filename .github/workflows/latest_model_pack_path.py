import subprocess
import sys
from glob import glob


def get_existing_model_pack_paths(model_packs_dir):
    model_pack_paths = glob(f"{model_packs_dir}/model_pack*", recursive=True)
    return model_pack_paths


def get_git_commit_hashes():
    full_cmd = subprocess.Popen(["awk", "{print $1}"],
                                stdout=subprocess.PIPE,
                                stdin=subprocess.Popen(["git", "log", "--format=oneline"],
                                                       stdout=subprocess.PIPE).stdout)

    return [line.decode("utf-8").strip() for line in full_cmd.stdout.readlines()]


def get_latest_model_pack_path(model_packs_dir):
    model_pack_paths = get_existing_model_pack_paths(model_packs_dir)
    model_pack_git_hashes = [path.split("_")[1] for path in model_pack_paths]
    model_pack_git_hashes_to_idx = {commit_hash: i
                                    for i, commit_hash in enumerate(model_pack_git_hashes)}

    for git_hash in get_git_commit_hashes():
        if git_hash in model_pack_git_hashes_to_idx:
            idx = model_pack_git_hashes_to_idx[git_hash]
            return model_pack_paths[idx]
    return model_pack_paths[-1]


if __name__ == "__main__":
    model_packs_dir = sys.argv[1]
    print(get_latest_model_pack_path(model_packs_dir))
