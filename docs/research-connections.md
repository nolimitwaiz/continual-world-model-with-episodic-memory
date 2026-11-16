# Connection to Human-Memory Research

The link is a research question: **How does the organization of experience affect what can
be remembered later?** This project tests a computational version of part of that question.
It does not establish a biological explanation or an endorsement by any laboratory.

## Event Boundaries

[Baldassano, Chen, and colleagues (2017)](https://pubmed.ncbi.nlm.nih.gov/28772125/)
studied event structure during continuous narrative perception and memory. Their findings
motivate asking whether meaningful event boundaries help organize stored experience.

Our completed comparison is narrower: a model receives simulated states and actions,
and the experimenter supplies event boundaries. It does not infer boundaries from brain
activity, reproduce cortical dynamics, or demonstrate learned segmentation.

## Connections Between Events

[Lee and Chen (2022)](https://doi.org/10.1038/s41467-022-31965-2) examined how the network
structure of naturalistic events relates to memory. This motivates a different question:
does memory benefit from preserving how events relate, beyond deciding where to split them?

The FilmFest work in this repository reproduced downstream behavioral statistics from
released source data. Its separate graph-retrieval diagnostic used the same predicted
graph to define the task and aid retrieval. That diagnostic is not independent evidence
that relationship-based memory improves recall.

## What the Current Experiments Contribute

The synthetic tests provide a controlled setting for changing memory organization. The
Sherlock analysis checks one text-retrieval setup using naturalistic annotations and
retains the negative main result. Together they identify a question for a better test:
is the benefit due to boundaries, preserved content, or how one memory connects to another?

They do not yet separate these explanations. The [short proposal](research-brief.md)
sets out a paired-history experiment and the controls needed to make that separation.
The [full results](results.md) keep development findings, reproduction, and exploratory
diagnostics distinct.

## A Useful Question for Discussion

What experiment would show that a model remembers who did what, in what order, and what
changed, rather than merely matching familiar people, objects, or words? Which existing
recall annotations would support that test, and which additional labels would need to be
collected independently?

This is an alignment of research interests, not evidence that the complete proposed world
model fits a particular laboratory's current projects or supervision capacity.
