import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'supergraph',
  tagline: 'Memory infrastructure for AI agents',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
    faster: true,
  },

  url: 'https://supergraph-docs.orkait.com',
  baseUrl: '/',

  organizationName: 'orkait',
  projectName: 'supergraph',

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          editUrl: 'https://github.com/orkait/supergraph/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          lastmod: 'date',
          changefreq: 'weekly',
          priority: 0.5,
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        indexBlog: false,
        docsRouteBasePath: '/',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    image: 'img/supergraph-social-card.png',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    metadata: [
      {name: 'keywords', content: 'supergraph, agent memory, vector database, graph database, SQLite, Python, DSL, retrieval'},
    ],
    navbar: {
      title: 'supergraph',
      logo: {
        alt: 'supergraph',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://pypi.org/project/supergraph/',
          label: 'PyPI',
          position: 'right',
        },
        {
          href: 'https://github.com/orkait/supergraph',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Intro', to: '/'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: 'https://github.com/orkait/supergraph'},
            {label: 'PyPI', href: 'https://pypi.org/project/supergraph/'},
            {label: 'Issues', href: 'https://github.com/orkait/supergraph/issues'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Orkait. Licensed AGPL-3.0.`,
    },
    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['python', 'bash', 'sql', 'json', 'toml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
