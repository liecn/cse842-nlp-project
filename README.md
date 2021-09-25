# cse842-nlp-project

The large-scale StackOverflow.com is an online question-and-answer site for programmers. This dataset includes an archive of Stack Overflow content, including posts, votes, tags, and badges. 

### Instruction

The data is divided into three sets:
- Train: Data before 2018-01-01 UTC except the held-out users. 342,477 unique users with 135,818,730 examples.
- Held-out: All examples from users with user_id % 10 == 0 (all dates). 38,758 unique users with 16,491,230 examples.
- Test: All examples after 2018-01-01 UTC except from held-out users. 204,088 unique users with 16,586,035 examples.

### Pre-processing

The data consists of the body text of all questions and answers. The bodies were parsed into sentences, and any user with fewer than 100 sentences was expunged from the data. Minimal preprocessing was performed as follows:

Lowercase the text,
Unescape HTML symbols,
Remove non-ascii symbols,
Separate punctuation as individual tokens (except apostrophes and hyphens),
Removing extraneous whitespace,
Replacing URLS with a special token.

In addition the following metadata is available:
Creation date
Question title
Question tags
Question score
Type ('question' or 'answer')

### Link
A detailed instruction about the dataset can be found at https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data 
Its original location is at https://storage.googleapis.com/tff-datasets-public/stackoverflow.tar.bz2.
