- Require `({})` for block expressions; they’re rare, so it shouldn’t be
  an issue (and it resolves the `match {` and `{} a` weirdness.
- Heterogeneous Ranges
- Named Parameters
- Make use of `PromoteMemOpInterface` and friends.
