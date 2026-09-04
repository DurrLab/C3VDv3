/**
 * Generic CSV-to-table loader — now supports an extra “Old Name” column
 * and the three Original C3VD tables.
 */
document.addEventListener('DOMContentLoaded', () => {
  const tables = [
    // C3VDv2 tables you already had
    { id: 'registered-videos-table',  csv: 'assets/data/C3VDv2_data_summary_registered.csv',              withOld: false },
    { id: 'deformation-videos-table', csv: 'assets/data/C3VDv2_data_summary_deformation.csv',  withOld: false },
    { id: 'screening-videos-table',   csv: 'assets/data/C3VDv2_data_summary_screening.csv',    withOld: false },
    { id: 'models-table',   csv: 'assets/data/C3VDv2_data_summary_models.csv',    withOld: false },

    // NEW: Original C3VD tables (CSVs you generated earlier)
    { id: 'orig-registered-table', csv: 'assets/data/C3VD_data_summary_registered.csv', withOld: true },
    { id: 'orig-screening-table',  csv: 'assets/data/C3VD_data_summary_screening.csv',  withOld: true },
    { id: 'orig-models-table',     csv: 'assets/data/C3VD_data_summary_models.csv',                    withOld: false } // model table has no Old Name
  ];

  tables.forEach(cfg => populateTable(cfg));
});

function populateTable({ id, csv, withOld }) {
  const tbody = document.querySelector(`#${id} tbody`);
  if (!tbody) return;

  // Check if this is a model table
  const isModelTable = id === 'models-table' || id === 'orig-models-table';
  const isOrigModelsTable = id === 'orig-models-table';

  fetch(csv)
    .then(r => r.ok ? r.text() : Promise.reject(`${csv} → ${r.status}`))
    .then(text => {
      if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);

      // CSV → array of rows
      const rows = text.trim().split(/\r?\n/).map(r => r.split(','));
      const headers = rows.shift().map(h => h.trim().toLowerCase());

      const idx = name => headers.indexOf(name.toLowerCase());

      rows.forEach(cols => {
        const cells = [];

        // For model tables: just render Colon and Segment, then generate downloads
        if (isModelTable) {
          const colon = cols[idx('colon')];
          const segment = cols[idx('segment')];

          cells.push(`<td>${colon}</td>`);
          cells.push(`<td>${segment}</td>`);

          const baseName = `${colon}_${segment}`.toLowerCase();
          const modelLabel = `${baseName}_model.zip`; 
          const isFullSegment = isOrigModelsTable && segment?.trim().toLowerCase() === 'full';
          const moldLabel = isFullSegment ? '' : `${baseName}_mold.zip`;

          cells.push(`<td><a href="#"><span>${modelLabel}</span></a></td>`);
          cells.push(
            moldLabel
              ? `<td><a href="#"><span>${moldLabel}</span></a></td>`
              : '<td></td>'
          );
        } else {

        headers.forEach((h, i) => {
          // 👉 Skip the CSV column called “Video Name”
          if (h === 'video name') return;
          else if (h === 'download') return;
          else if (h === 'preview') {
            const ytId     = cols[i]; // e.g. "esEBiCIfDUY"
            cells.push(
              `<td data-youtube="${ytId}" class="preview-cell" style="cursor:pointer;">
                 <span style="text-decoration:underline;">Preview</span>
               </td>`
            );
          // } else if (h === 'download') {
          //   // CSV supplies the full href
          //   const href = cols[i] || '#';
          //   const vidName = cols[idx('video name')]?.toLowerCase() ?? '';
          //   const label     = `${vidName}.zip`;
          //   cells.push(`<td><a href="${href}">${label}</a></td>`);
          } else {
            cells.push(`<td>${cols[i]}</td>`);
          }
        });

        const vidName = cols[idx('Video Name')]?.toLowerCase() ?? '';
        const label     = `${vidName}.zip`;
        cells.push(`<td><a href="#">${label}</a></td>`);

        if (withOld && idx('old name') === -1) {
          // Add blank Old-Name cell so column count matches header
          cells.push('<td></td>');
        }
      }
        tbody.insertAdjacentHTML('beforeend', `<tr>${cells.join('')}</tr>`);
      });

      // Add console message about finished loading tables
      console.log(`Finished loading table: ${id} from ${csv}`);
      
      if (typeof window.initDataverseDownloadLinks === "function") {
        window.initDataverseDownloadLinks();
      } else {
        console.error("initDataverseDownloadLinks is not defined on window");
      }
      console.log("initDataverseDownloadLinks on window?", window.initDataverseDownloadLinks);
    })
    .catch(err => console.error('CSV load error for', id, err));
}
