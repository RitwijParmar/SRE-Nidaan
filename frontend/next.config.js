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
    return [];
  },

  async rewrites() {
    const bodyBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    return [
      {
        source: "/api/:path*",
        destination: `${bodyBase}/api/:path*`,
      },
      {
        source: "/body/:path*",
        destination: `${bodyBase}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
