/*
 * Chart B — Per-team horizontal bar chart.
 * For the national team chosen in the dropdown, counts players by the country
 * of their club, sorted descending. Home (club_country == national_team) bars
 * are visually distinguished from abroad bars.
 *
 * Requires D3 v7 (loaded via _includes/head-includes.html when `d3: true`).
 * Mount points in the post:
 *   <select id="wc2026-team-select"></select>
 *   <div id="wc2026-team-bars"></div>
 */
(function () {
  "use strict";
  var MOUNT = "wc2026-team-bars";
  var SELECT = "wc2026-team-select";
  var DATA_URL = "/assets/data/wc2026_squads.json";
  var DEFAULT_TEAM = "Qatar";

  var COLOR_HOME = "#59a14f";   // plays in their own country
  var COLOR_ABROAD = "#4e79a7"; // plays abroad

  function countsForTeam(rows, team) {
    var byCountry = d3.rollup(
      rows.filter(function (r) { return r.national_team === team; }),
      function (v) { return v.length; },
      function (r) { return r.club_country; }
    );
    return Array.from(byCountry, function (kv) {
      return { country: kv[0], count: kv[1], home: kv[0] === team };
    }).sort(function (a, b) { return d3.descending(a.count, b.count) || d3.ascending(a.country, b.country); });
  }

  function render(container, data, team) {
    var width = Math.max(280, container.clientWidth);
    var rowH = 26;
    var margin = { top: 8, right: 36, bottom: 34, left: 132 };
    var height = margin.top + margin.bottom + data.length * rowH;

    container.innerHTML = "";

    var svg = d3.select(container).append("svg")
      .attr("viewBox", [0, 0, width, height])
      .attr("width", "100%")
      .attr("height", "auto")
      .attr("role", "img")
      .attr("aria-labelledby", MOUNT + "-title " + MOUNT + "-desc");
    svg.append("title").attr("id", MOUNT + "-title")
      .text("Club countries for " + team + " squad players");
    svg.append("desc").attr("id", MOUNT + "-desc")
      .text("Horizontal bar chart counting " + team + " players by the country of their club, "
        + "sorted from most to fewest. Home-country bars are coloured differently from abroad bars.");

    var x = d3.scaleLinear()
      .domain([0, d3.max(data, function (d) { return d.count; }) || 1]).nice()
      .range([margin.left, width - margin.right]);
    var y = d3.scaleBand()
      .domain(data.map(function (d) { return d.country; }))
      .range([margin.top, height - margin.bottom])
      .padding(0.18);

    // Bars
    svg.append("g").selectAll("rect")
      .data(data)
      .join("rect")
      .attr("class", "bar")
      .attr("x", x(0))
      .attr("y", function (d) { return y(d.country); })
      .attr("width", function (d) { return x(d.count) - x(0); })
      .attr("height", y.bandwidth())
      .attr("fill", function (d) { return d.home ? COLOR_HOME : COLOR_ABROAD; })
      .append("title")
      .text(function (d) {
        return d.country + ": " + d.count + " player" + (d.count === 1 ? "" : "s")
          + (d.home ? " (home)" : " (abroad)");
      });

    // Value labels at bar ends
    svg.append("g").selectAll("text.val")
      .data(data)
      .join("text")
      .attr("class", "bar-value")
      .attr("x", function (d) { return x(d.count) + 4; })
      .attr("y", function (d) { return y(d.country) + y.bandwidth() / 2; })
      .attr("dy", "0.35em")
      .text(function (d) { return d.count; });

    // Y axis (real tick elements)
    svg.append("g")
      .attr("class", "axis axis-y")
      .attr("transform", "translate(" + margin.left + ",0)")
      .call(d3.axisLeft(y).tickSizeOuter(0));

    // X axis (integer ticks)
    var maxV = d3.max(data, function (d) { return d.count; }) || 1;
    svg.append("g")
      .attr("class", "axis axis-x")
      .attr("transform", "translate(0," + (height - margin.bottom) + ")")
      .call(d3.axisBottom(x).ticks(Math.min(maxV, 8)).tickFormat(d3.format("d")).tickSizeOuter(0));
  }

  function init() {
    var container = document.getElementById(MOUNT);
    var select = document.getElementById(SELECT);
    if (!container || !select || typeof d3 === "undefined") { return; }

    d3.json(DATA_URL).then(function (rows) {
      var teams = Array.from(new Set(rows.map(function (r) { return r.national_team; }))).sort();
      select.innerHTML = "";
      teams.forEach(function (t) {
        var opt = document.createElement("option");
        opt.value = t; opt.textContent = t;
        if (t === DEFAULT_TEAM) { opt.selected = true; }
        select.appendChild(opt);
      });
      var current = teams.indexOf(DEFAULT_TEAM) >= 0 ? DEFAULT_TEAM : teams[0];

      function draw() { render(container, countsForTeam(rows, current), current); }
      draw();

      select.addEventListener("change", function () { current = select.value; draw(); });
      if (window.ResizeObserver) {
        var t;
        new ResizeObserver(function () {
          clearTimeout(t);
          t = setTimeout(draw, 150);
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
