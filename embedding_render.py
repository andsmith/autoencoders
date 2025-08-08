"""
Class for rendering views of the embedding space and
interacting with it:
   * show a visually clean representation of the embedding
   * 


Usage:   

      de = DigitEmbedding(embedding_function, images, locations, labels, min_dist_ratio=1.0)

      frame = de.render_embedding(img_size, view_bbox)

      frame, sampled_points = de.render_interpolated_path(frame, start, end, density=0.5)

      indices = de.nearest_neighbors(point, n=3)

      indices_in_hull = de.inside_hull(point_list)


"""
