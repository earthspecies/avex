document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.sidebar-tree li.has-children > a.reference').forEach(function (link) {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      var checkbox = link.parentElement.querySelector('input.toctree-checkbox');
      if (checkbox) checkbox.checked = !checkbox.checked;
    });
  });
});
