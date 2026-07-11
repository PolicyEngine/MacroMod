// Scroll-reveal for bands: adds `.in-view` to elements as they enter the
// viewport, matching the `.reveal` / `.in-view` CSS in style.css. Split out
// of field.js so pages without the hero canvas (every strategy and paper
// page) still get their content revealed — field.js exits early via
// `if (!canvas) return`, which used to skip this observer entirely on any
// page without a `#field` canvas, leaving `.band-head`, `.prose`, `.cards`,
// and `.metrics` stuck at opacity: 0 forever.
(function () {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const targets = document.querySelectorAll(
    ".band-head, .prose, .cards, .metrics, .caveat, .commons-rails, .stack-note, .band-cta h2, .band-cta .hero-cta"
  );
  if (reduce || !("IntersectionObserver" in window)) {
    targets.forEach(function (el) { el.classList.add("in-view"); });
    return;
  }
  const io = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add("in-view");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  targets.forEach(function (el) { io.observe(el); });
})();
