from huggingface_hub import snapshot_download

data_dir = "/mnt/data/vhoangth2"

# snapshot_download(
#     repo_id="fangqi/WMPO",
#     repo_type="model",
#     local_dir=f"{data_dir}/ckpts/WMPO",
#     local_dir_use_symlinks=False,
#     # allow_patterns=["checkpoint_files/**"]
#     # allow_patterns=["checkpoint_files/SFT_models/square/**"]
#     # allow_patterns=["checkpoint_files/world_models/square/P_128/**"]
#     # allow_patterns=["checkpoint_files/reward_models/**square**"]
#     # allow_patterns=["checkpoint_files/**/OpenSora-STDiT-v3/**"]
#     # allow_patterns=["checkpoint_files/**/vae/**"]
#     # allow_patterns=["checkpoint_files/**/OpenX_pretrained/**"]
# )

snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=f"{data_dir}/datasets/WMPO",
    local_dir_use_symlinks=False,
    # allow_patterns=["data_files/**"]
    # allow_patterns=["data_files/**/square_d0_300_demos/**"]
    # allow_patterns=["data_files/example_rollouts/square_128_demos/**"]
    allow_patterns=["data_files/statistics/square/**"]
)