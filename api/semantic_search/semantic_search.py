from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch
import numpy as np
import ownphotos
from torch import Tensor, device
# from api.models.photo import Photo

dir_clip_ViT_B_32_model = ownphotos.settings.CLIP_ROOT

class SemanticSearch():
    model_is_loaded = False

    def load(self):
        self.load_model()
        model_is_loaded = True
        pass

    def unload(self):
        self.model = None
        model_is_loaded = False
        pass

    def load_model(self):
        self.model = SentenceTransformer(dir_clip_ViT_B_32_model)

    # def load_embeddings(self):
    #     embeddings = []
        
    #     for obj in Photo.objects.all():
    #         emb = obj.clip_embeddings
    #         if emb:
    #             embeddings.append(np.array(emb))
            

    #     self.embeddings = np.array(embeddings)

    def calculate_clip_embeddings(self, img_paths):
        if not self.model_is_loaded:
            self.load()

        if type(img_paths) is list:
            imgs = list(map(Image.open, img_paths))
        else:
            imgs = [Image.open(img_paths)]

        imgs_emb = self.model.encode(imgs, batch_size=32, convert_to_tensor=True)

        if type(img_paths) is list:
            magnitudes = map(np.linalg.norm, imgs_emb)

            return imgs_emb, magnitudes
        else:
            img_emb = imgs_emb[0].tolist()
            magnitude = np.linalg.norm(img_emb)

            return img_emb, magnitude

    def calculate_query_embeddings(self, query):
        if not self.model_is_loaded:
            self.load()

        query_emb = self.model.encode([query], convert_to_tensor=True)[0].tolist()
        magnitude = np.linalg.norm(query_emb)

        return query_emb, magnitude

    def semantic_search(self,
                        query_embeddings: Tensor,
                        corpus_embeddings: Tensor,
                        query_chunk_size: int = 100,
                        corpus_chunk_size: int = 10000,
                        top_k: int = 10,
                        score_function: Callable[[Tensor, Tensor], Tensor] = self.cos_sim):
        """
        This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param query_embeddings: A 2 dimensional tensor with the query embeddings.
        :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
        :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
        :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
        :param top_k: Retrieve top k matching entries.
        :param score_function: Funtion for computing scores. By default, cosine similarity.
        :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
        """

        if isinstance(query_embeddings, (np.ndarray, np.generic)):
            query_embeddings = torch.from_numpy(query_embeddings)
        elif isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)

        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
            corpus_embeddings = torch.from_numpy(corpus_embeddings)
        elif isinstance(corpus_embeddings, list):
            corpus_embeddings = torch.stack(corpus_embeddings)


        #Check that corpus and queries are on the same device
        if corpus_embeddings.device != query_embeddings.device:
            query_embeddings = query_embeddings.to(corpus_embeddings.device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarites
                cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx+query_chunk_size], corpus_embeddings[corpus_start_idx:corpus_start_idx+corpus_chunk_size])

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

        #Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list

    def cos_sim(self, a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


semantic_search_instance = SemanticSearch()