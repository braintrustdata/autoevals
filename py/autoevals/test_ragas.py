import asyncio

from pytest import approx

from autoevals.ragas import *

data = {
    "input": "Can starred docs from different workspaces be accessed in one place?",
    "output": "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
    "expected": "Yes, all starred docs, even from multiple different workspaces, will live in the My Shortcuts section.",
    "context": [
        "Not all Coda docs are used in the same way. You'll inevitably have a few that you use every week, and some that you'll only use once. This is where starred docs can help you stay organized.\n\n\n\nStarring docs is a great way to mark docs of personal importance. After you star a doc, it will live in a section on your doc list called **[My Shortcuts](https://coda.io/shortcuts)**. All starred docs, even from multiple different workspaces, will live in this section.\n\n\n\nStarring docs only saves them to your personal My Shortcuts. It doesn\u2019t affect the view for others in your workspace. If you\u2019re wanting to shortcut docs not just for yourself but also for others in your team or workspace, you\u2019ll [use pinning](https://help.coda.io/en/articles/2865511-starred-pinned-docs) instead."
    ],
}


def test_ragas_retrieval():
    metrics = [
        (ContextEntityRecall(), 0.5),
        (ContextRelevancy(), 0.7),
        (ContextRecall(), 1),
        (ContextPrecision(), 1),
    ]

    for m, score in metrics:
        sync_score = m.eval(**data).score
        async_score = asyncio.run(m.eval_async(**data)).score

        if score == 1:
            assert sync_score == score
            assert async_score == score
        else:
            assert sync_score >= score
            assert async_score >= score
