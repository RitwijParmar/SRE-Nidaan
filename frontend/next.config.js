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
    const bodyBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    const usesRemoteBody = /^https?:\/\//.test(bodyBase) && !bodyBase.includes("localhost");
    if (!usesRemoteBody) {
      return [];
    }
    return [
      {
        source: "/api/:path*",
        destination: `${bodyBase}/api/:path*`,
        permanent: false,
      },
    ];
  },

  async rewrites() {
    const bodyBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
    const usesRemoteBody = /^https?:\/\//.test(bodyBase) && !bodyBase.includes("localhost");
    if (usesRemoteBody) {
      return [
        {
          source: "/body/:path*",
          destination: `${bodyBase}/:path*`,
        },
      ];
    }
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
