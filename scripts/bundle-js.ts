import * as esbuild from 'esbuild';

esbuild.build({
    entryPoints: ['js/index.ts'],
    bundle: true,
    outfile: 'jsdist/bundle.js',
    platform: 'neutral',
    target: 'es2017',
    format: 'esm',
    sourcemap: true,
    loader: {
        '.yaml': 'text'
    },
    packages: 'external'
});
