import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

generator = T5ForConditionalGeneration.from_pretrained('pretrained_models/t5-large')
tokenizer = AutoTokenizer.from_pretrained('pretrained_models/t5-large')
query = 'Generating explanations for question with candidates: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? bank, library department store, mall, new york explanation:'
query = tokenizer([query], return_tensors='pt')

ret_dict = generator.generate(
            input_ids=query['input_ids'],
            attention_mask=query['attention_mask'],
            num_return_sequences=5,
            num_beams=10,
            max_length=128 + 2,
            # +2 from original because we start at step=1 and stop before max_length
            min_length=0 + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

print(ret_dict.keys())
decoder_hidden_states = ret_dict["decoder_hidden_states"]
print(len(decoder_hidden_states))
print(len(decoder_hidden_states[0]))
print(decoder_hidden_states[0][-1].shape)  # [beam, 1, h]

beam_indices = ret_dict['beam_indices']
print(beam_indices)
