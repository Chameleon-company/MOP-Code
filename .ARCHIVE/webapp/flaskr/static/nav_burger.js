// Occurs when the nav-burger is clicked
document.addEventListener('DOMContentLoaded', function () {

    var navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
    if (navbarBurgers.length > 0) {
      // add onclick event for each item in the navbar
      navbarBurgers.forEach(function (item) {
        item.addEventListener('click', function () {
          
          var target = item.dataset.target;
          var target = document.getElementById(target);
          
          item.classList.toggle('is-active');
          target.classList.toggle('is-active');
        });
      });
    }
  });