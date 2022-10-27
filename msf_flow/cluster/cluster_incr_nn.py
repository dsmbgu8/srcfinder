import os
import json
import argparse
import itertools
from csv import DictReader, DictWriter
from math import sqrt
import numpy as np
from utm import from_latlon
from pyclustering.cluster import cluster_visualizer

if 'AWS' in os.environ.keys() and os.environ['AWS'] == 'TRUE':
    print('in AWS')
    import boto3

def parse_args():
    """Retrieve command line parameters.

    Returns:
        ArgumentParse: command line parameters
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("plumes", help="path to input plume file")
    parser.add_argument("output", help="path to output plume file")
    parser.add_argument("-r", "--radius", required=True,
                        help="radius (meters) for nearest neighbor clustering")
    parser.add_argument("-v", "--visualize", action='store_true',
                        help="Show plot of points/clusters (default=no plot)")
    args = parser.parse_args()
    return args.plumes, args.output, float(args.radius), args.visualize

def latlon2utm (coords_latlon):
    coords_utm = [from_latlon(*coord)[0:2] for coord in coords_latlon]
    return coords_utm

def read_plumes(plume_fname,
                lat_key="Plume Latitude (deg)",
                lon_key="Plume Longitude (deg)",
                fill_value="-9999"):
    with open(plume_fname, 'r') as fin:
        reader = DictReader(fin, skipinitialspace=True)
        plume_list = [d for d in reader
                      if ((d[lat_key] != fill_value) and
                          (d[lon_key] != fill_value))]
    return plume_list

def get_plume_coords(plume_list,
                     lat_key="Plume Latitude (deg)",
                     lon_key="Plume Longitude (deg)",
                     fill_value="-9999"):
    coords_latlon = [[float(d[lat_key]), float(d[lon_key])] for d in plume_list
                         if ((d[lat_key] != fill_value) and
                             (d[lon_key] != fill_value))]
    return coords_latlon

def source_num_to_str(source_num, prefix=""):
    return "{}{:08d}".format(prefix, source_num)

def calc_sq_dist(coord1, coord2):
    assert (len(coord1) == len(coord2)), "Cannot calculate distance between two coordinates with different dimensions!"
    dist = 0
    for i in range(len(coord1)):
        diff = coord2[i] - coord1[i]
        dist += diff * diff
    return dist

def update_clusters_single_sample(i, sample, sources, clusters, source_mapping,
                                  radius):
    sq_dists = [calc_sq_dist(sample[i], source) for source in sources]
    ind_min = np.argmin(sq_dists)
    min_dist = sqrt(sq_dists[ind_min])
    if min_dist > radius:
        # None of the current clusters are close enough.  Create a new
        # cluster centered on this sample.
        sources.append(sample[i])
        clusters.append([i])
        source_id = len(sources)
    else:
        # Add this sample to the minimum distance cluster and update
        # cluster coordinate.
        clusters[ind_min].append(i)
        sources[ind_min] = np.mean([sample[ind]
                                    for ind in clusters[ind_min]], axis=0)
        source_id = ind_min + 1
    source_id_str = source_num_to_str(source_id, prefix="A")
    source_mapping[i] = source_id_str

def cluster(sample, radius=100):
    source_mapping = {}
    source_id = 1
    source_id_str = source_num_to_str(source_id, prefix="A")
    sources = [sample[0]]
    source_mapping[0] = source_id_str
    clusters = [[0]]
    for i in range(1, len(sample)):
        update_clusters_single_sample(i, sample, sources, clusters,
                                      source_mapping, radius)
    return source_mapping, clusters

def main(AWS=False, event={}):
    # default params
    plume_fname = ''
    out_fname = 'cluster_out.csv'
    radius = 150
    do_vis = False

    if AWS:
        key = event['filename']
        plume_fname = '/tmp/'+key
        out_fname = plume_fname.replace('plumes_ext', 'plumes_cluster')

        plume_dir = '/'.join(plume_fname.split('/')[:-1])
        if not os.path.exists(plume_dir):
            os.makedirs(plume_dir)
        out_dir = '/'.join(out_fname.split('/')[:-1])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # download the plume file
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(event['bucket'])
        print('download {} to {}'.format(key, plume_fname))
        bucket.download_file(key, plume_fname)
    else:
        # Parse command line arguments.
        plume_fname, out_fname, radius, do_vis = parse_args()

    # Load plume coordinates.
    plume_list = read_plumes(plume_fname)

    # Compute UTM coordinates of the plumes.
    sample = latlon2utm(get_plume_coords(plume_list))

    # Perform cluster analysis.
    source_mapping, clusters = cluster(sample, radius=radius)

    # Add source id to each plume.
    plume_list_with_source = [plume_list[i].update({"Source ID": source_mapping[i]})
                              or plume_list[i]
                              for i in range(len(plume_list))]

    # Get field names from the first plume
    field_names = list(plume_list[0].keys())

    print('write output to {}'.format(out_fname))
    # Write processed plumes to output file
    with open(out_fname, 'w') as fout:
        writer = DictWriter(fout, fieldnames=field_names)
        writer.writeheader()
        for plume in plume_list_with_source:
            writer.writerow(plume)
    print("Plume file with source identification written to {}".
          format(out_fname))

    # upload output to AWS
    if AWS:
        out_key = '/'.join(out_fname.split('/')[2:])
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(event['bucket'])
        print('upload {} to {}'.format(out_fname, out_key))
        bucket.upload_file(out_fname, out_key)

    # Visualize clustering results.
    if do_vis:
        visualizer = cluster_visualizer()
        visualizer.append_clusters(clusters, sample)
        visualizer.show()

def lambda_handler(event, context):
    bucket = event['Records'][0]["s3"]["bucket"]["name"]
    filename = event['Records'][0]["s3"]["object"]["key"]
    event = {'bucket':bucket, 'filename':filename}
    main(AWS=True, event=event)
    return {
        'statusCode': 200,
        'body': json.dumps('Done clustering plumes and adding source ids!')
    }

if __name__ == "__main__":
    main()
