# CHANGELOG.md

## (2020-21-11)

**New:**

- Configuration json updated
- Experimenting with augmentations
- Enriched computational graph
- Interfaces are now organised
- Utilities for keras training callbacks, several training utils and plots

## (2020-27-10)

**New:**

- Images from unsplash from new unsplash release have been accumulated.
- Dataset exploration, import new dataset can handle new releases in datasets.
- New datasets are available, square images(total dataset) & horizontal images
- Training, inference & model architectures are available
- **interfaces** folder contains endpoints for image transformations and tfrecord generation
- **configuration** is used to setup various in project settings
- **utils** for data cleaning, plots, gpu initialization & training
- **trained_models** saving-loading

**Fix:**
- Horizontal images dataset fixed due to inconsistencies in image size.
- Image resize transformation

**Data:**

- Dataset is released in tf records:
    - *square contains the total lite dataset free from nan values (21445 images). Images are resized and transformed in 1:1 format meaning that both horizontal and vertical shots are now square images, filled with 0pad pixels.

        **train**: 19301, **validation**: 1072, **test**: 1072, **image resolution**: 400x400.
    - *horizontal contains only horizontal images transformed in 3:2 format (cropped/0padded) from lite dataset, free from nan values (11425 images)
        
        **train**: 10283, **validation**: 571, **test**: 571, **image resolution**: 300x200



## (2020-3-10)

**New:**
- Restructure folder tree
- Adopt oo paterns and modules

**Fix:**
- Dataset exploration contained git diffs

**Data:**
- [#33gitissue](https://github.com/unsplash/datasets/issues/33) 318 new photos added with Unsplash v1.1.0 integration, since same number of images have been deleted and replaced with new ones to their platform.
- Updated unsplashed-resized and unsplashed-horizontal datasets in gdrive

**Integrity checks (SHA-256):**

- tf-records-square.zip: `894c7605344a87e958770817297fdc64a9d7c2937531522ea5356a0c231562e0`
- tf-records-horizontal.zip: 
`8171a5ab98847515711f880daa598c91f258662d852f03c03462b7522e8aac57`
- unsplash-square.zip: `d1bec7f428fe0146f2a15e30206621a2f02c1be2977d6eb46479bcefb004d511`
- unsplash-horizontal.zip: `ab14062fe1d90546927c58c8510bd4f515d1b295728818ffec83499837185c96`