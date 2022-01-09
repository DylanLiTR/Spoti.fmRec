window.onload = function () {
	// Add event listener for when the user submits their LastFM username
	document.getElementById("form").addEventListener('submit', function (event) {
		// Prevent the page from refreshing
		event.preventDefault();
		var recs = document.getElementById("recs")
		if (recs) {
			recs.remove();
		}
		
		var main = document.getElementById("main");
		var loader = document.createElement("div");
		loader.className = "loader";
		main.appendChild(loader);
		
		const formData = new FormData(form);
		fetch('/', {
			method: 'POST',
			body: formData,
		}).then(async function (response) {
			var recs = document.createElement("div");
			recs.id = "recs";
			recs.className = "middle";
			
			recs.innerHTML = await response.text();
			
			main.removeChild(main.lastChild);
			main.appendChild(recs);
		})
	})
}