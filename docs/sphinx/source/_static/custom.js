// Custom sidebar navigation behavior for sphinx_rtd_theme
// Makes expand/collapse only trigger on arrow click, not link click

document.addEventListener("DOMContentLoaded", function() {
    
    // Process all toctree links that have expandable children
    document.querySelectorAll('.wy-menu-vertical .toctree-l1, .wy-menu-vertical .toctree-l2, .wy-menu-vertical .toctree-l3').forEach(function(li) {
        const link = li.querySelector(':scope > a');
        const ul = li.querySelector(':scope > ul');
        
        // Only process items that have children (expandable)
        if (!link || !ul) return;
        
        // Create a toggle button for expand/collapse
        const toggle = document.createElement('span');
        toggle.className = 'nav-toggle';
        toggle.setAttribute('role', 'button');
        toggle.setAttribute('aria-label', 'Toggle submenu');
        
        // Insert toggle after the link text
        link.appendChild(toggle);
        
        // Toggle expand/collapse on arrow click
        toggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            li.classList.toggle('current');
        });
        
        // Link click should only navigate, not toggle
        link.addEventListener('click', function(e) {
            // If clicking on the toggle, don't navigate
            if (e.target === toggle || toggle.contains(e.target)) {
                e.preventDefault();
                return;
            }
            // Otherwise, allow normal navigation (don't prevent default)
        });
    });
});
