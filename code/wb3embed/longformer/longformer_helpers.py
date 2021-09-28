
import torch
import numpy as np


def chunk_tokens(input_ids, token_limit=4096):

    token_seq_len = len(input_ids[0])
    input_ids_chunked = torch.LongTensor()

    if token_seq_len > token_limit:

        long_input_ids = input_ids[0]
        # sent_ends = torch.where(long_input_ids == 4, long_input_ids) ## check for sentence end
        n_chunks = int(np.ceil(token_seq_len / token_limit))
        input_id_chunks = torch.chunk(long_input_ids, n_chunks)
        chunk_seq_len = int(np.ceil(token_seq_len / n_chunks))

        ## add start and end tokens
        for input_id_chunk in input_id_chunks:

            if input_id_chunk[0] != 0:
                input_id_chunk[0] = 0  ## start token <s>
            if input_id_chunk[-1] != 2:
                input_id_chunk[-1] = 2  ## end token </s>

            if input_id_chunk.shape[0] < chunk_seq_len:  ## pad chunk sequence length to chunk_seq_len
                padding = torch.ones(chunk_seq_len - input_id_chunk.shape[0], dtype=torch.long)
                input_id_chunk = torch.cat((input_id_chunk, padding), 0)  ## pad token <pad>

            input_id_chunk = input_id_chunk.unsqueeze(0)
            input_ids_chunked = torch.cat((input_ids_chunked, input_id_chunk), 0)

    else:
        input_ids_chunked = input_ids

    return input_ids_chunked



def build_attention_mask(input_id_chunks, attention_pool_tokens = None):
    attention_mask = torch.LongTensor()
    global_attention_mask = torch.LongTensor()

    for input_id_chunk in input_id_chunks:
        attention_mask_chunk = torch.ones(input_id_chunk.shape[0], dtype=torch.long) ## local attention on all tokens
        global_attention_mask_chunk = torch.zeros(input_id_chunk.shape[0], dtype=torch.long)  ## global attention only on attention_pool_tokens

        for attention_pool_token in attention_pool_tokens:
            global_attention_mask_chunk[input_id_chunk == attention_pool_token] = 1 #1  ## global attention on attention_pool_tokens

        ## local attention
        attention_mask_chunk = attention_mask_chunk.unsqueeze(0)
        attention_mask = torch.cat((attention_mask, attention_mask_chunk), 0)

        ## global attention
        global_attention_mask_chunk = global_attention_mask_chunk.unsqueeze(0)
        global_attention_mask = torch.cat((global_attention_mask, global_attention_mask_chunk), 0)

    return attention_mask, global_attention_mask




if __name__ == "__main__":

    pass
    #input_ids = chunk_tokens()
    #attention_mask = build_attention_mask(input_ids, attention_tokens)

