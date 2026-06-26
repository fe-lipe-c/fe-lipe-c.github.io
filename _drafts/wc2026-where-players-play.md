---
layout: post
title: "Where World Cup 2026 Players Come From and Where They Play"
author: Felipe Costa
date: 2026-06-22
d3: true
---

<!--
  DRAFT. Charts are wired up; prose marked TODO is for me to write.
  Data: assets/data/wc2026_squads.json (1,246 players; 2 "Unattached" rows
  with plays_abroad = N/A were dropped). Continent buckets: see
  assets/data/continent_lookup.json (geographic framing, 7 continents).
-->

The 2026 FIFA World Cup is getting closer to its final group's matches, 

A note on grouping: each country is assigned to one of seven continents
(Europe, Africa, N. America, S. America, W. Asia, E. Asia, Oceania) on a
geographic basis. A few cross-boundary cases were judgement calls — Turkey,
the South Caucasus, and Central Asia sit in W. Asia, while Russia and Cyprus
sit in Europe. See the lookup table in the repo for the full mapping.

## The big picture: continent-to-continent flows

The chord diagram below is **directed**. Each arc is a continent. A ribbon
leaves the continent a player represents (their national team) and points,
arrow-first, to the continent where their club is based. Thickness is the
number of players. Ribbons that loop back to their own arc are players who
stay on their home continent.

<div id="wc2026-chord" class="d3-chart"
     aria-label="Directed chord diagram of player flows between continents"></div>

<!-- TODO commentary — hero chord. Talking points to expand:
  - "Africa -> Europe": the dominant export ribbon; far more African players
    are employed in Europe than at home.
  - "Europe stays home": Europe's largest ribbon loops back to itself.
  - "Asia plays at home": W. Asia and E. Asia are mostly self-contained.
  - N. and S. America: a meaningful share still flows to Europe.
  Write the actual analysis here.
-->

*TODO: write the analysis of the aggregate flows (Africa to Europe, Europe
staying home, Asia playing at home).*

## One team at a time

The aggregate hides how different individual squads are. Pick a national team
below to see how its 26 players split across club countries, sorted from most
to fewest. Green bars are players based **at home**; blue bars are players
based **abroad**.

<div class="wc2026-controls">
  <label for="wc2026-team-select">National team:</label>
  <select id="wc2026-team-select" aria-label="Choose a national team"></select>
</div>

<div class="wc2026-legend" aria-hidden="true">
  <span><span class="swatch swatch-home"></span>Plays at home</span>
  <span><span class="swatch swatch-abroad"></span>Plays abroad</span>
</div>

<div id="wc2026-team-bars" class="d3-chart"
     aria-label="Bar chart of club countries for the selected national team"></div>

<!-- TODO commentary — per-team chart. Talking points to expand:
  - NOTE / FLAG: the brief called Qatar the "all-abroad" case, but in this
    data Qatar is the opposite — 25 of 26 players are at home in the Qatar
    Stars League (1 abroad, in Spain). The genuine all-ABROAD illustration is
    Morocco (24 of 26 abroad). Decide which to feature; to switch the default,
    change DEFAULT_TEAM in assets/js/wc2026-team-bars.js.
  - Contrast a domestic-heavy squad (Saudi Arabia, England) with an
    export-heavy one (Morocco, Senegal).
  Write the actual analysis here.
-->

*TODO: write the per-team analysis. (See the note in the source comment about
the Qatar vs. Morocco framing before finalising the default.)*

<script src="/assets/js/wc2026-chord.js"></script>
<script src="/assets/js/wc2026-team-bars.js"></script>
