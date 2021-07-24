
import operator
from functools import reduce

from rest_framework.compat import distinct
from django.db.models.expressions import RawSQL
from django.db.models import Q
from rest_framework import filters
from api.semantic_search.semantic_search import semantic_search_instance

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

            conditions.append(reduce(operator.or_, queries))
        queryset = queryset.filter(reduce(operator.and_, conditions))

        if self.must_call_distinct(queryset, search_fields):
            # Filtering against a many-to-many field requires us to
            # call queryset.distinct() in order to avoid duplicate items
            # in the resulting queryset.
            # We try to avoid this if possible, for performance reasons.
            queryset = distinct(queryset, base)
        return queryset