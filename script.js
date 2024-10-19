// JavaScript to handle slider movement
const slider = document.getElementById('slider');
const originalPanel = document.getElementById('originalVideo').parentElement;
const reconstructedPanel = document.getElementById('reconstructedVideo').parentElement;

slider.addEventListener('mousedown', function(e) {
    document.addEventListener('mousemove', resizePanels);
    document.addEventListener('mouseup', () => {
        document.removeEventListener('mousemove', resizePanels);
    });
});

function resizePanels(e) {
    const windowWidth = window.innerWidth;
    const leftWidth = e.clientX / windowWidth * 100;
    const rightWidth = 100 - leftWidth;

    originalPanel.style.width = ${leftWidth}%;
    reconstructedPanel.style.width = ${rightWidth}%;
}

// Dismiss alert functionality
const dismissButtons = document.querySelectorAll('.dismiss');
dismissButtons.forEach(button => {
    button.addEventListener('click', function() {
        this.parentElement.style.display = 'none';
    });
});