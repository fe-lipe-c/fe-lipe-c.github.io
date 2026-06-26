/*
 * Chart A — Directed continent-to-continent chord diagram.
 * Origin arc = continent of the player's national_team.
 * Destination (arrow head) = continent of the player's club_country.
 * Aggregate "where players come from vs. where they play" view.
 *
 * Requires D3 v7 (loaded via _includes/head-includes.html when `d3: true`).
 * Mount point: a <div id="wc2026-chord"> in the post. No inline data; the
 * cleaned squad JSON is fetched from /assets/data/.
 */
(function () {
  "use strict";
  var MOUNT = "wc2026-chord";
  var DATA_URL = "/assets/data/wc2026_squads.json";

  // Fixed arc order + palette (shared visual language across the post).
  var CONTINENTS = ["Europe", "Africa", "N. America", "S. America", "W. Asia", "E. Asia", "Oceania"];
  var COLOR = d3.scaleOrdinal()
    .domain(CONTINENTS)
    .range(["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2", "#9c755f"]);

  function buildMatrix(rows) {
    var idx = {};
    CONTINENTS.forEach(function (c, i) { idx[c] = i; });
    var m = CONTINENTS.map(function () { return CONTINENTS.map(function () { return 0; }); });
    rows.forEach(function (r) {
      var o = idx[r.origin_continent], d = idx[r.dest_continent];
      if (o != null && d != null) { m[o][d] += 1; }
    });
    return m;
  }

  function render(container, matrix) {
    var width = Math.max(280, container.clientWidth);
    var height = width;                 // square
    var outerR = Math.min(width, height) * 0.5 - 96;
    var innerR = outerR - 14;

    container.innerHTML = "";

    var svg = d3.select(container).append("svg")
      .attr("viewBox", [-width / 2, -height / 2, width, height])
      .attr("width", "100%")
      .attr("height", "auto")
      .attr("role", "img")
      .attr("aria-labelledby", MOUNT + "-title " + MOUNT + "-desc");
    svg.append("title").attr("id", MOUNT + "-title")
      .text("Directed chord diagram of player flows between continents");
    svg.append("desc").attr("id", MOUNT + "-desc")
      .text("Each arc is one of seven continents. Ribbons run from the continent of a "
        + "player's national team to the continent where their club is based; arrow heads "
        + "point to the destination. Ribbon thickness is the number of players.");

    var chord = d3.chordDirected()
      .padAngle(12 / innerR)
      .sortSubgroups(d3.descending)
      .sortChords(d3.descending);
    var chords = chord(matrix);

    var arc = d3.arc().innerRadius(innerR).outerRadius(outerR);
    var ribbon = d3.ribbonArrow().radius(innerR - 1).headRadius(innerR / 12);

    // Arc groups (one per continent)
    var group = svg.append("g")
      .attr("role", "list")
      .attr("aria-label", "Continents")
      .selectAll("g")
      .data(chords.groups)
      .join("g")
      .attr("role", "listitem");

    group.append("path")
      .attr("class", "chord-arc")
      .attr("fill", function (d) { return COLOR(CONTINENTS[d.index]); })
      .attr("stroke", function (d) { return d3.rgb(COLOR(CONTINENTS[d.index])).darker(); })
      .attr("d", arc)
      .append("title")
      .text(function (d) { return CONTINENTS[d.index] + ": " + d.value + " players (outgoing)"; });

    // Arc labels
    group.append("text")
      .each(function (d) { d.angle = (d.startAngle + d.endAngle) / 2; })
      .attr("class", "chord-label")
      .attr("dy", "0.35em")
      .attr("transform", function (d) {
        return "rotate(" + (d.angle * 180 / Math.PI - 90) + ")"
          + "translate(" + (outerR + 8) + ")"
          + (d.angle > Math.PI ? "rotate(180)" : "");
      })
      .attr("text-anchor", function (d) { return d.angle > Math.PI ? "end" : null; })
      .text(function (d) { return CONTINENTS[d.index]; });

    // Directed ribbons, coloured by source continent
    svg.append("g")
      .attr("fill-opacity", 0.72)
      .selectAll("path")
      .data(chords)
      .join("path")
      .attr("class", "chord-ribbon")
      .attr("d", ribbon)
      .attr("fill", function (d) { return COLOR(CONTINENTS[d.source.index]); })
      .attr("stroke", function (d) { return d3.rgb(COLOR(CONTINENTS[d.source.index])).darker(); })
      .append("title")
      .text(function (d) {
        return CONTINENTS[d.source.index] + " → " + CONTINENTS[d.target.index]
          + ": " + d.source.value + " players";
      });
  }

  function init() {
    var container = document.getElementById(MOUNT);
    if (!container || typeof d3 === "undefined") { return; }
    d3.json(DATA_URL).then(function (rows) {
      var matrix = buildMatrix(rows);
      render(container, matrix);
      if (window.ResizeObserver) {
        var t;
        new ResizeObserver(function () {
          clearTimeout(t);
          t = setTimeout(function () { render(container, matrix); }, 150);
        }).observe(container);
      }
    }).catch(function (e) {
      container.textContent = "Could not load chart data.";
      if (window.console) { console.error(e); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
