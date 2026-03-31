import arxiv
client = arxiv.Client()
search = arxiv.Search(
    query="quantum machine learning",
    max_results=200,
    sort_by=arxiv.SortCriterion.Relevance
)

print(search.results())