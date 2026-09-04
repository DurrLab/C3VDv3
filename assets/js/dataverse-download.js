// Dataverse download handler
console.log("Dataverse download script loaded");
// One-time metadata cache
let dataverseMetadata = null;
// Track if event listener has been added
let clickHandlerAttached = false;

window.initDataverseDownloadLinks = async function () {
  console.log("Dataverse download script running");

  const DOMAIN = "https://archive.data.jhu.edu";
  const DOI = "10.7281/T1/JC64MK";

  async function fetchMetadataOnce() {
    if (dataverseMetadata) return dataverseMetadata;
    console.log("Fetching dataset metadata...");

    try {
      const apiUrl = `${DOMAIN}/api/datasets/:persistentId?persistentId=doi:${DOI}`;
      
      // Try local proxy first (if running), then fallback to public proxies
      const proxies = [
        `https://api.cors.lol/?url=${encodeURIComponent(apiUrl)}`,
        `https://api.codetabs.com/v1/proxy/?quest=${encodeURIComponent(apiUrl)}`
      ];
      
      let response;
      let lastError;
      
      for (const proxyUrl of proxies) {
        try {
          console.log(`Trying proxy: ${proxyUrl.substring(0, 60)}...`);
          response = await fetch(proxyUrl, {
            headers: {
              'Accept': 'application/json',
            }
          });
          
          if (response.ok) {
            const text = await response.text();
            // Some proxies wrap the response, try to parse as JSON
            let data;
            try {
              data = JSON.parse(text);
            } catch (e) {
              // If parsing fails, the proxy might have returned the data directly
              // Try to extract JSON from the response
              const jsonMatch = text.match(/\{[\s\S]*\}/);
              if (jsonMatch) {
                data = JSON.parse(jsonMatch[0]);
              } else {
                throw new Error('Could not parse JSON from proxy response');
              }
            }
            
            dataverseMetadata = data.data?.latestVersion?.files || [];
            console.log(`Fetched ${dataverseMetadata.length} files from Dataverse.`);
            return dataverseMetadata;
          }
        } catch (proxyError) {
          console.warn(`Proxy failed:`, proxyError.message);
          lastError = proxyError;
          continue;
        }
      }
      
      throw lastError || new Error('All proxy attempts failed');
    } catch (err) {
      console.error("Failed to fetch metadata:", err);
      dataverseMetadata = []; // Prevent retry flood
    }

    return dataverseMetadata;
  }

  // Use event delegation to catch all clicks on download links, even dynamically created ones
  // This ensures links are intercepted even if they're created after this function runs
  // Only attach the listener once
  if (!clickHandlerAttached) {
    clickHandlerAttached = true;
    document.addEventListener('click', async function(e) {
    // Handle case where target might be a text node or child element
    let link = e.target;
    // If target is not an anchor, try to find the closest anchor
    if (link.nodeName !== 'A') {
      link = link.closest('a');
    }
    if (!link || link.nodeName !== 'A') return;
    
    const href = link.getAttribute('href');
    // Check if this is a download link we should intercept
    const isDownloadLink = href === '#' || 
                          (href && (href.includes('.csv') || href.includes('.zip') || href.includes('.xlsx') || href.includes('/assets/data/')));
    
    if (!isDownloadLink) return;
    
    console.log('Download link clicked:', href, link.textContent);
    
    // Prevent default navigation
    e.preventDefault();
    e.stopPropagation();
    
    // Extract filename
    const filenameSpan = link.querySelector('span');
    const filenameText = filenameSpan ? filenameSpan.textContent.trim() : link.textContent.trim();
    const filename = filenameText.split(' ')[0];
    
    console.log('Extracted filename:', filename);
    
    if (!filename) {
      console.warn('No filename found for link:', link);
      return; // Skip if no filename found
    }
    
    const originalHTML = link.innerHTML;
    link.innerHTML = '<span style="color: black;">Loading...</span>';
    link.style.pointerEvents = 'none';

    try {
      const files = await fetchMetadataOnce();
      const file = files.find(f => f.label === filename);

      if (!file) {
        console.error(`File "${filename}" not found in metadata`);
        alert(`File "${filename}" not found in dataset.`);
        link.innerHTML = originalHTML;
        link.style.pointerEvents = 'auto';
        return;
      }

      const fileId = file.dataFile.id;
      const downloadUrl = `${DOMAIN}/api/access/datafile/${fileId}`;
      const tempLink = document.createElement('a');
      tempLink.href = downloadUrl;
      tempLink.download = filename;
      tempLink.target = '_blank';
      document.body.appendChild(tempLink);
      tempLink.click();
      document.body.removeChild(tempLink);
    } catch (err) {
      console.error(`Error downloading file "${filename}":`, err);
      alert(`Error downloading file "${filename}": ${err.message}`);
    } finally {
      link.innerHTML = originalHTML;
      link.style.pointerEvents = 'auto';
    }
    }, true); // Use capture phase to catch before other handlers
  }

  // Also process existing links to update their hrefs and styling
  const downloadLinks = document.querySelectorAll('a[href="#"]:not(.download-processed), a[href*=".csv"]:not(.download-processed), a[href*=".zip"]:not(.download-processed), a[href*=".xlsx"]:not(.download-processed), a[href*="/assets/data/"]:not(.download-processed)');
  console.log(`Found ${downloadLinks.length} download links to process`);

  downloadLinks.forEach(link => {
    // Update href to "#" to prevent navigation if it's a local file path
    const originalHref = link.getAttribute('href');
    if (originalHref && originalHref !== '#' && (originalHref.startsWith('/') || originalHref.startsWith('./') || originalHref.includes('.csv') || originalHref.includes('.zip') || originalHref.includes('.xlsx'))) {
      link.setAttribute('data-original-href', originalHref);
      link.href = '#';
    }

    // Mark as processed
    link.classList.add('download-processed');

    // Style
    const filenameSpan = link.querySelector('span');
    if (filenameSpan) {
      link.title = "Click to download";
      link.style.cursor = "pointer";
    }
  });
};

// Auto-initialize: Call immediately for static links
if (window.initDataverseDownloadLinks) {
    window.initDataverseDownloadLinks();
}

// Also call after dynamic content loads (CSV tables)
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        if (window.initDataverseDownloadLinks) {
            window.initDataverseDownloadLinks();
        }
    }, 1000);
});

// Call after window load as well
window.addEventListener('load', function() {
    setTimeout(function() {
        if (window.initDataverseDownloadLinks) {
            window.initDataverseDownloadLinks();
        }
    }, 500);
});
