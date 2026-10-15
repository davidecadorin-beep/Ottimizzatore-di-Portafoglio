(function () {
  var tapeInner = document.getElementById('ticker-tape-inner');
  if (!tapeInner) return;

  var lastData = [];

  function renderItem(item) {
    var changeSign = item.change_pct > 0 ? '+' : '';
    var colorClass = item.direction === 'up' ? 'tape-up' :
                     (item.direction === 'down' ? 'tape-down' : 'tape-flat');
    var arrow = item.direction === 'up' ? '▲' : (item.direction === 'down' ? '▼' : '—');
    var rateStr;
    if (item.symbol === 'BTC/USD' || item.symbol === 'ETH/USD') {
      rateStr = Number(item.rate).toLocaleString('it-IT', {minimumFractionDigits: 0, maximumFractionDigits: 0});
    } else {
      rateStr = Number(item.rate).toFixed(4);
    }
    return '<span class="tape-item">' +
      '<span class="tape-symbol">' + item.symbol + '</span>' +
      '<span class="tape-rate">' + rateStr + '</span>' +
      '<span class="tape-change ' + colorClass + '">' + arrow + ' ' + changeSign + item.change_pct.toFixed(2) + '%</span>' +
    '</span>';
  }

  function renderTape(data) {
    if (!data || !data.length) return;
    lastData = data;
    var html = data.map(renderItem).join('') + data.map(renderItem).join('');
    tapeInner.innerHTML = html;
  }

  function fetchAndUpdate() {
    fetch('/api/fx')
      .then(function(r) { return r.json(); })
      .then(renderTape)
      .catch(function() {
        if (lastData.length) renderTape(lastData);
      });
  }

  fetchAndUpdate();
  setInterval(fetchAndUpdate, 60000);
})();
