python colorize_dataset.py \
  --out_dir        out \
  --dataset        filipino/tagalog_filipino_eng_translation          \
  --split          val                        \
  --num_tokens     2048                       \
  --device         cuda:0                     \
  --block_size     256                        \
  --output_file    val_colour.txt                # save ANSI text (opt.)

# drwxrwxr-x 2 kauna kauna      4096 May 21 11:47 20250521_114704

# /media/kauna/e70770fd-475d-43ec-b3e8-57f3bef7890d1/2025/nanogpt_new_datasets/out master* ⇡
# nanogpt 11:56❯ p train.py  --max_sample_tokens 256 --colorize_mode all --colorize_output --max_iters 5000 --dataset filipino/tagalog_filipino_eng_translation --out_dir test_color --sample_start_tokens "mabuhay, pupunta sa bahay"


