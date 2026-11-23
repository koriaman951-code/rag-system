import arxiv

# Direct client search (works on modern arxiv 2.x+)
client = arxiv.Client()
search = arxiv.Search(
    query="large language model",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance
)

for i, result in enumerate(client.results(search)):
    print(f"\nResult {i+1}")
    print("Title:", result.title)
    print("Authors:", [author.name for author in result.authors])
    print("Published:", result.published)
    print("Summary:", result.summary[:500])
    