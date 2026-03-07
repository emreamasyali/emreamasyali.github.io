# Personal Website — Emre Amasyalı

Source files for my personal academic website built with [Quarto](https://quarto.org).

## Structure

- `index.qmd` — Home page with bio and links
- `about.qmd` — Extended academic bio
- `publications.qmd` — List of publications
- `teaching.qmd` — Courses taught
- `contact.qmd` — Contact information
- `_quarto.yml` — Site configuration
- `styles.css` — Custom CSS
- `docs/` — Compiled website output (served by GitHub Pages)
- `files/profile/` — Profile photo

## Building

1. Install [Quarto](https://quarto.org/docs/get-started/)
2. Render the site: `quarto render`
3. Preview locally: `quarto preview`

## Deployment (GitHub Pages)

The site renders to the `docs/` directory. To publish on GitHub Pages:

1. Push this repository to GitHub (recommended repo name: `emreamasyali.github.io`)
2. Go to **Settings → Pages**
3. Set source to **Deploy from a branch**, branch `main`, folder `/docs`
4. The site will be live at `https://emreamasyali.github.io`

See the Quarto publishing guide: <https://quarto.org/docs/publishing/github-pages.html>
