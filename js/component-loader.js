// Component Loader for Philip Juenemann Portfolio
// Automatically loads and injects components into pages

class ComponentLoader {
    constructor() {
        this.components = new Map();
    }

    async loadComponent(componentName, containerId) {
        try {
            const response = await fetch(`../components/${componentName}.html`);
            if (!response.ok) {
                throw new Error(`Failed to load component: ${componentName}`);
            }
            const html = await response.text();

            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = html;
                this.components.set(componentName, html);

                // Set active navigation state
                if (componentName === 'sidebar') {
                    this.setActiveNavItem();
                }
            }
        } catch (error) {
            console.error(`Error loading component ${componentName}:`, error);
        }
    }

    setActiveNavItem() {
        const currentPage = this.getCurrentPage();
        const navLinks = document.querySelectorAll('.nav-link[data-page]');

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-page') === currentPage) {
                link.classList.add('active');
            }
        });
    }

    getCurrentPage() {
        const path = window.location.pathname;
        if (path.includes('/proj0/')) return 'proj0';
        if (path.includes('/proj1/')) return 'proj1';
        if (path.includes('/proj2/')) return 'proj2';
        return 'portfolio';
    }

    async loadProjectHeader(projectData) {
        await this.loadComponent('header-project', 'project-header');

        // Update project-specific data
        const titleElement = document.querySelector('[data-project-title]');
        const breadcrumbElement = document.querySelector('[data-breadcrumb-current]');
        const publishDateElement = document.querySelector('[data-publish-date]');
        const descriptionElement = document.querySelector('[data-project-description]');

        if (titleElement) titleElement.textContent = projectData.title;
        if (breadcrumbElement) breadcrumbElement.textContent = projectData.breadcrumb;
        if (publishDateElement) publishDateElement.textContent = projectData.publishDate;
        if (descriptionElement) descriptionElement.textContent = projectData.description;
    }

    async loadPortfolioHeader() {
        await this.loadComponent('header-portfolio', 'portfolio-header');
    }

    async loadSidebar() {
        await this.loadComponent('sidebar', 'sidebar-container');
    }

    async loadFooter() {
        await this.loadComponent('footer', 'footer-container');
    }
}

// Initialize component loader
const componentLoader = new ComponentLoader();

// Auto-load components when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Load sidebar and footer on all pages
    await componentLoader.loadSidebar();
    await componentLoader.loadFooter();

    // Load appropriate header based on page type
    const currentPage = componentLoader.getCurrentPage();

    if (currentPage === 'portfolio') {
        await componentLoader.loadPortfolioHeader();
    } else {
        // Load project header with default data
        // Individual pages can override this data
        await componentLoader.loadProjectHeader({
            title: 'Project Title',
            breadcrumb: 'Project',
            publishDate: 'Date',
            description: 'Project description'
        });
    }
});

// Export for use in individual pages
window.ComponentLoader = ComponentLoader;
window.componentLoader = componentLoader;
