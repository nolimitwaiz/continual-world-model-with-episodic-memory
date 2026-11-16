#!/usr/bin/env node
// Optional PNG rendering for inspection and PDF export. Requires the sharp package.
const fs = require("node:fs");
const path = require("node:path");
const { parseArgs } = require("node:util");
const sharp = require("sharp");
const { values } = parseArgs({
  options: {
    output: { type: "string", default: ".preview/figures" },
    names: { type: "string" }
  }
});
const root = path.resolve(__dirname, "..");
const source = path.join(root, "docs/figures");
const output = path.resolve(root, values.output);
const names = values.names ? values.names.split(",") :
  fs.readdirSync(source).filter(name => name.endsWith(".svg")).map(name => name.slice(0, -4));
fs.mkdirSync(output, { recursive: true });
(async () => {
  for (const name of names) {
    if (!/^[a-z0-9-]+$/.test(name)) throw new Error("Invalid figure name");
    await sharp(path.join(source, name + ".svg"), { density: 144 })
      .png().toFile(path.join(output, name + ".png"));
  }
  console.log(JSON.stringify({ rendered_figures: names.length, training_run: false }));
})().catch(error => { console.error(error.message); process.exitCode = 1; });
