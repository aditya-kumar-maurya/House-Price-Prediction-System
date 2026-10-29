document.addEventListener("DOMContentLoaded", () => {
  const quickSearch = document.getElementById("quickSearch");
  if (quickSearch) {
    quickSearch.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && quickSearch.value.trim()) {
        window.location.href = "/houses?q=" + encodeURIComponent(quickSearch.value.trim());
      }
    });
  }
  document.querySelectorAll(".flash").forEach((item) => {
    setTimeout(() => { if (item.isConnected) item.style.opacity = "0.96"; }, 4000);
  });
});
