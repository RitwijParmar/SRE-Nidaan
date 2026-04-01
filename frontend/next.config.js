/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: "/",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, max-age=0",
          },
        ],
      },
    ];
  },

  async redirects() {
    const canonicalHost = process.env.NEXT_PUBLIC_CANONICAL_HOST;
    const aliasHost = process.env.NEXT_PUBLIC_BIND_ALIAS_HOST;

    if (!canonicalHost || !aliasHost) {
      return [];
    }

    return [
      {
        source: "/:path*",
        has: [{ type: "host", value: aliasHost }],
        destination: `https://${canonicalHost}/:path*`,
        permanent: false,
      },
    ];
  },

  async rewrites() {
    const bodyBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    return [
      {
        source: "/body/:path*",
        destination: `${bodyBase}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
