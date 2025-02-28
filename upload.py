import asyncio
from dev.upload_to_vectordb import VectorDBUploader

async def main():
    # Initialize the VectorDBUploader with the sitemap URL and the sitemap flag
    sitemap_url = "https://cps.northeastern.edu/cps-program-sitemap.xml"  # Replace with your desired URL
    use_sitemap = True  # Set to False if you want to process a single URL

    uploader = VectorDBUploader(url=sitemap_url, sitemap=True,format_content=True,source="cps_program_docs")
    await uploader.run()

if __name__ == "__main__":
    asyncio.run(main())