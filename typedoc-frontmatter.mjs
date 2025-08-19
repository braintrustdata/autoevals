import { MarkdownPageEvent } from "typedoc-plugin-markdown";

export function load(app) {
  app.renderer.on(MarkdownPageEvent.BEGIN, (page) => {
    if (page.model?.name) {
      page.frontmatter = {
        title: page.model?.name,
        ...page.frontmatter,
      };
    }
  });
}
