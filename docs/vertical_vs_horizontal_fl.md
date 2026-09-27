# Vertical vs horizontal federated learning

Horizontal FL generally partitions rows/entities while parties share a similar feature schema. Vertical FL aligns the same or overlapping entities across parties that own different feature columns.

VertiMosaic implements the vertical setting. The benchmark never relabels row-partitioned training as VFL.
