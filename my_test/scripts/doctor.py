#!/usr/bin/env python3
"""Readiness audit: explicitly separates local integration and native gameplay."""
import argparse, json, pathlib, shutil, socket, sys
p=argparse.ArgumentParser()
p.add_argument('--library')
p.add_argument('--resource-path',default=None)
p.add_argument('--check-dns',action='store_true')
a=p.parse_args()
checks={'python':sys.version.split()[0]}
for name in ('git','meson','ninja','gcc','cmake'):
    checks['binary_'+name]=shutil.which(name) or None
if a.check_dns:
    try: checks['github_dns']=socket.getaddrinfo('github.com',443)[0][4][0]
    except Exception as e: checks['github_dns_error']=str(e)
if a.library:
    checks['native_library_exists']=pathlib.Path(a.library).is_file()
    if checks['native_library_exists']:
        try:
            sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
            from native_rl.taisei_ffi import NativeTaisei
            with NativeTaisei(a.library,resource_path=a.resource_path) as sim: checks['native_loaded_and_abi_verified']=True
        except Exception as e:checks['native_load_error']=str(e)
print(json.dumps(checks,indent=2))
if a.library and not checks.get('native_loaded_and_abi_verified'):
    sys.exit(1)
