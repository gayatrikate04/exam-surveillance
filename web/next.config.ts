import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Video uploads go through /detection-api rewrites; default proxy buffer is 10MB.
  experimental: {
    proxyClientMaxBodySize: "200mb",
  },
  async rewrites() {
    return [
      {
        source: "/detection-api/:path*",
        destination: "http://127.0.0.1:8000/:path*",
      },
    ];
  },
};

export default nextConfig;
