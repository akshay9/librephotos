import numpy as np
import operator
import torch
from functools import reduce

from rest_framework.compat import distinct
from django.db.models.expressions import RawSQL
from django.db.models import Q
from rest_framework import filters
from sentence_transformers import util
from api.semantic_search.semantic_search import semantic_search_instance
from api.models.photo import Photo

torch.set_num_threads(4)

class SemanticSearchFilter(filters.SearchFilter):
    def filter_queryset(self, request, queryset, view):
        search_fields = self.get_search_fields(view, request)
        search_terms = self.get_search_terms(request)

        if not search_fields or not search_terms:
            return queryset

        orm_lookups = [
            self.construct_search(str(search_field))
            for search_field in search_fields
        ]

        if request.user.semantic_search_topk > 0:
            query = request.query_params.get('search')
            emb, magnitude = semantic_search_instance.calculate_query_embeddings(query)

            # if not hasattr(self, "corpus_img_hash") or len(self.corpus_img_hash) == 0:
            #     self.corpus_img_hash = []
            #     self.corpus_emb = []

            #     # corpus = np.array([[obj.image_hash, np.array(obj.clip_embeddings)] for obj in Photo.objects.filter(owner_id=request.user.id)])
            #     for obj in Photo.objects.filter(owner_id=request.user.id):
            #         self.corpus_img_hash.append(obj.image_hash)
            #         self.corpus_emb.append(np.array(obj.clip_embeddings))

            #     self.corpus_img_hash = np.array(self.corpus_img_hash)
            #     self.corpus_emb = np.array(self.corpus_emb)
            
            # res = util.semantic_search(np.array([emb]), self.corpus_emb, top_k=request.user.semantic_search_topk)
            # corpus_ids = [img["corpus_id"] for img in res[0]]
            # print(self.corpus_img_hash[corpus_ids])

            semantic_search_instance.unload()

            # Calculating the cosine similarity
            semantic_search_query = """
                Select image_hash from (SELECT (
                    SELECT sum(a*b)/(%s*clip_embeddings_magnitude)
                    FROM unnest(
                    clip_embeddings, -- ex1
                    %s  -- ex2
                    ) AS t(a,b)
                ) as similarity, *
                    FROM public.api_photo
                WHERE
                    owner_id=%s
                Order by similarity desc
                Limit %s) as t
                where t.similarity > 0.25
            """

        base = queryset
        conditions = []
        for search_term in search_terms:
            queries = [
                Q(**{orm_lookup: search_term})
                for orm_lookup in orm_lookups
            ]

            if request.user.semantic_search_topk > 0:
                queries += [Q(image_hash__in=RawSQL(semantic_search_query, [magnitude, emb, request.user.id, request.user.semantic_search_topk]))]
                # queries += [Q(image_hash__in=self.corpus_img_hash[corpus_ids])]
                pass

            conditions.append(reduce(operator.or_, queries))
        queryset = queryset.filter(reduce(operator.and_, conditions))

        if self.must_call_distinct(queryset, search_fields):
            # Filtering against a many-to-many field requires us to
            # call queryset.distinct() in order to avoid duplicate items
            # in the resulting queryset.
            # We try to avoid this if possible, for performance reasons.
            queryset = distinct(queryset, base)
        return queryset