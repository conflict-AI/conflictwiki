from wb0configs import configs
from wb0configs.helpers import load_file
from wb3embed.longformer.longformer_helpers import chunk_tokens, build_attention_mask

import re
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig


class WikiLongformer():

    def __init__(self, entity_dict = {}):
        self.tokenizer, self.model, self.device = self.load_tokenizer_model()
        self.add_entities_to_tokenizer(entity_dict)


    def load_tokenizer_model(self, ):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        print(f"load_tokenizer_model running on {str(device)}")


        model_config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', attention_window=512, output_attentions=False, gradient_checkpointing=True, attention_mode = 'sliding_chunks')
        model = LongformerModel(model_config)
        model = model.to(device)

        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        tokenizer.model_max_length = model.config.max_position_embeddings

        return tokenizer, model, device


    def add_entities_to_tokenizer(self, entity_dict = {}):

        if len(entity_dict.keys()) > 0:
            print("added new entity tokens to tokenizer")
            self.token_id_entity_id = dict()
            token_id = self.tokenizer._convert_token_to_id("<mask>")

            for entity_id in entity_dict.keys():
                token_id += 1
                self.tokenizer.add_tokens(["<" + str(entity_id) + "/>"])
                self.token_id_entity_id[token_id] = entity_id

            self.model.resize_token_embeddings(len(self.tokenizer))  ## adjust model
        else:
            print("no entity tokens added to tokenizer")
            self.token_id_entity_id = entity_dict


    ### operative functions____________________________________________________________________


    def model_encode(self, input_id_chunks, attention_masks, global_attention_masks):

        input_attention = zip(input_id_chunks, attention_masks, global_attention_masks)
        tokenid_embed_dict = defaultdict(torch.Tensor)

        #for input_id, attention_mask in tqdm(input_attention):
        for input_id, attention_mask, global_attention_mask in input_attention:

            with torch.no_grad():
                outputs = self.model(input_ids=input_id.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), global_attention_mask = global_attention_mask.unsqueeze(0))

            sequence_output = outputs.last_hidden_state
            # pooled_output = outputs.pooler_output

            tokenid = input_id[global_attention_mask == 1]  ## get all ids with global_attention_masks
            embed = sequence_output[0, global_attention_mask == 1, :]
            tokenid_embed = zip(tokenid, embed.to(self.device))

            for tokenid, embed in tokenid_embed:
                embed = embed.unsqueeze(0)
                tokenid_embed_dict[int(tokenid)] = torch.cat((tokenid_embed_dict[int(tokenid)], embed),0)  ## concat all representations


        ## take mean if one token has multiple embeds and change token_id back to entity_id
        id_embed_dict = defaultdict(torch.Tensor)
        for tokenid, embed in tokenid_embed_dict.items():
            entity_id = self.tokenizer.convert_ids_to_tokens(int(tokenid))  ## token_id to entity_id

            if entity_id == "<s>": ## CLS token
                entity_id = 0
            else:
                entity_id = int(re.sub(r"(<)(.+?)(/>)", r'\2', entity_id))  ## remove < /> to get entity_id only
            id_embed_dict[entity_id] = torch.mean(embed, 0)

        return id_embed_dict


    def tokenize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)  # cls_token='<s>' 0, mask_token = <mask> 50264
        return inputs["input_ids"]


    def chunk_tokens(self, input_ids):

        input_ids_chunked = chunk_tokens(input_ids)
        attention_pool_tokens = list(self.token_id_entity_id.keys()) + [0] ## entity tokens + CLS token
        attention_mask_chunked, global_attention_mask_chunked = build_attention_mask(input_ids_chunked, attention_pool_tokens)
        return input_ids_chunked, attention_mask_chunked, global_attention_mask_chunked


    def text2embed(self, text):

        input_ids = self.tokenize(text)
        input_ids, attention_mask, global_attention_mask = self.chunk_tokens(input_ids)

        input_ids = input_ids.to(self.device)  ## to cuda
        attention_mask = attention_mask.to(self.device)  ## to cuda
        global_attention_mask = global_attention_mask.to(self.device)  ## to cuda

        id_embed = self.model_encode(input_ids, attention_mask, global_attention_mask)
        return id_embed



if __name__ == "__main__":

    config = configs.ConfigBase()
    entity_dict = load_file(config.get_path("entity_retrieval") / "entity_id_name", ftype="pkl")
    #text = ' '.join(['Hello world! Ok'] * 3410)
    text = ('On 7 October 2001, in response to the September 11 attacks, President of the <3434750/> George W. Bush announced that airstrikes targeting <1921/> and the <30635/> had begun in <737/>. <22738/> primarily refers to the War in <737/>, but it is also affiliated with counterterrorism operations in other countries, such as OEF-<23440/> and OEF-Trans Sahara.After 13 years, on 28 December 2014, President Barack Obama announced the end of <22738/> in <737/>.')

    #wikilongformer = WikiLongformer()
    wikilongformer = WikiLongformer(entity_dict)

    ## do on each section
    input_ids = wikilongformer.tokenize(text)
    input_ids_chunked, attention_mask_chunked = wikilongformer.chunk_tokens(input_ids)
    id_embed_dict = wikilongformer.model_encode(input_ids_chunked, attention_mask_chunked)
    print(id_embed_dict.keys())

