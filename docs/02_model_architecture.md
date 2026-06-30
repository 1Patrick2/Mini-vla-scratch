# Model Architecture

The first model target is:

```text
image -> CNN vision encoder -> image feature
text -> Embedding + GRU language encoder -> text feature
state -> MLP state encoder -> state feature
concat -> fusion MLP -> action head -> action
```
