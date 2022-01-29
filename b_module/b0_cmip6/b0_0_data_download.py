

# =============================================================================
# region download data via ESGF and OPeNDAP

# http://gallery.pangeo.io/repos/pangeo-gallery/cmip6/search_and_load_with_esgf_opendap.html


def esgf_search(
        server="https://esgf-node.llnl.gov/esg-search/search",
        files_type="OPENDAP", local_node=True, project="CMIP6",
        verbose=False, format="application%2Fsolr%2Bjson",
        use_csrf=False, **search):
    
    import requests
    
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"] = "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken
    
    payload["format"] = format
    
    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k, d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    
    return sorted(all_files)


'''
# examples
import xarray as xr
xr.set_options(display_style='html')

result = esgf_search(
    activity_id='CMIP', table_id='Amon', variable_id='tas',
    experiment_id='historical', institution_id="NCAR",
    source_id="CESM2", member_id="r10i1p1f1")
ds = xr.open_mfdataset(result[0:4], combine='by_coords')

'''



# endregion
# =============================================================================


