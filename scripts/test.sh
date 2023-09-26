python scripts/benchmark.py \
       --model facebook/opt-6.7b  \
       --assist-model facebook/opt-125m\
       --tokenizer facebook/opt-125m\
       --len-out 128 \
       --speculate-step 0 1 2 3 4 5 6 7 8\
       --batch-size 1 2 4 8 16 32\
       --fp16 \
       --dataset alespalla/chatbot_instruction_prompts \
       --dataset-truncate 200 \
       # --collect-stats
