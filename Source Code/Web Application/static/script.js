window.onload = function () {
  document.getElementById("loading").style.display = "none";
};

const myButton = document.getElementById('buttonHome');
myButton.addEventListener('click', function () {
  document.getElementById("loading").style.display = "block";
});


function Applying(route) {
  window.location.href = route;
}
