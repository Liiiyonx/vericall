// 前端体检：提取页内 $("id")/getElementById 引用，与 HTML id= 对比，报告悬空引用
const fs = require('fs');
['login', 'index', 'child', 'elder'].forEach(n => {
  const h = fs.readFileSync('web/' + n + '.html', 'utf8');
  const ids = new Set([...h.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
  const refs = new Set();
  [...h.matchAll(/\$\("([^"]+)"\)/g)].forEach(m => refs.add(m[1]));
  [...h.matchAll(/getElementById\("([^"]+)"\)/g)].forEach(m => refs.add(m[1]));
  // 动态拼接的 ID（如 btnId+"T"）忽略
  const missing = [...refs].filter(r => !ids.has(r) && !r.includes('+') && !r.includes('"'));
  console.log('== ' + n + '.html ==');
  console.log('  ids=' + ids.size + ' refs=' + refs.size + ' missing=' + (missing.length ? missing.join(', ') : 'none'));
});
