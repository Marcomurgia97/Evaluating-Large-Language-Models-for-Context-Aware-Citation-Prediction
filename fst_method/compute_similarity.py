from sentence_transformers import SentenceTransformer, util
import torch


class SimilarityComputer:
    def __init__(self, model):
        self.model = SentenceTransformer(model, device='cuda')

    def get_emeddings(self, lists_of_titles):
        embeddings = self.model.encode(lists_of_titles, convert_to_tensor=True)
        return embeddings

    def get_most_similar(self, search, titles):
        #top_k = min(1, len(titles))
        top_k = min(len(titles), len(titles))

        emb1 = self.get_emeddings([search])
        emb2 = self.get_emeddings(titles)

        cos_scores = util.cos_sim(emb1, emb2)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return top_results



