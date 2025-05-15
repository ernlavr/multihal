# multihal

# Improvements.
- Use context paragraph to extract additional subjects
- If wikipedia link is available, derive the entity and add it to the list of subjects


## Problematic cases
- halueval_2812

- halueval_6876 (qa)
Q: Are New York Woman and New Idea, from Australia, both weekly publications?
A: no

- halubench - 097d72ee-5615-4cdb-a324-dad475fca91d (general)
Q: Was there a greater decrease in polio from 1953 to 1957, or 1957 to 1961?

## Reasoning cases
halueval_572
Knocked Up starred the actor and filmmaker of what dual nationality?
Ans: American Canadian
Paths: 
Seth Rogen -> citizen -> USA
Seth Rogen -> citizen -> Canadian
how could these be collapsed, e.g. 
Seth Rogen -> citizen -> USA | Canadian?


## yes/no questions
1. use only subjects, maybe falcon properties matching

## Evaluation
- Fleiss-Kappa (multi-rater agreemenet)
  - All scores
  - Binarized pass/fail scores (all)
  - Max scores per question


test