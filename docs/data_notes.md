# Data and policy notes

The source files are pinned to commit `7e74398ef4d9fec262bd7f64573d4d2692db2ead` and verified by SHA-256 before use.
Raw CSVs are never committed.

## Important catalogue facts

1. This is a catalogue, not behavior data. TMDb `popularity`, `vote_count`, and `vote_average` are global aggregates.
2. `show_id` collides across files (397 IDs, including movie/TV `18823`), so IDs are `movie_<id>` and `tv_<id>`.
3. Nine TV IDs are duplicated; the highest-popularity row is kept. The result is 31,991 unique items.
4. Movie and TV genre vocabularies differ. `Action & Adventure`, `Sci-Fi & Fantasy`, `War & Politics`, and `Kids` are
   expanded using the committed mapping; `TV Movie` and `Unknown` add no genre.
5. `rating` exactly duplicates the 0-10 vote average; it is not a maturity rating. `duration` is unusable and removed.
6. A zero rating/vote count means unrated. Ranking uses a Bayesian weighted rating rather than raw score or a global vote
   floor, which would disproportionately remove TV and recent titles.
7. `date_added` behaves as release date. No simulated event can precede it.
8. Exactly 1,000 titles per type/year indicates a sample. Popularity strongly favors recent content and differs by type,
   so a within-type percentile is used.
9. Metadata is sparse: 3,206 TV rows lack descriptions; 1,081 rows lack genres; 506 have neither and are not recommendable.
10. Duplicate normalized title/year responses are removed even when different IDs exist.
11. Movies alone contain budget/revenue; TV receives nulls in the unioned schema.

## Content-policy limitations

There is no maturity/adult flag. The implemented layers are:

1. bounded adult-content keywords over title and description;
2. low-evidence movie + Romance/Drama + fewer than 50 votes + zero budget;
3. a curated item-ID denylist.

The full reviewed run blocked 2,043 rows and left 29,443 recommendable. Manual top-20 inspection for demo users A/B/C
added `movie_829557` and `movie_829560` to the denylist.

This heuristic has known misses such as *Scarlet Innocence* and false positives such as *Dhoom Dhaam* and *Sidelined: The
QB and Me*. Mature mainstream titles can pass. Educational anatomy titles can also rank. The output is **not family-safe**
and must not be marketed as such. Production needs licensed age/maturity, regional availability, and content-rights feeds,
plus policy review and an appeal process.

## Simulator interpretation

Five thousand seeded users draw genre preferences around ten declared personas and sample without replacement from
affinity, language boost, and the twelfth power of within-type popularity percentile. Reserved users A/B/C represent
Action, Korean Romance/Drama, and Documentary. Their output validates system wiring, not human preference quality.

The measured 205k events, median 29/user, and 17.1% top-1% item share are sanity checks. Metrics should never be compared
to a production recommender benchmark or used to claim business lift.
