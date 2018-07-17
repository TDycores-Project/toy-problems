function n = compute_left_normal_vector(v1, v2)

  vec_from_1_to_2 = v2 - v1;
  
  n = [-vec_from_1_to_2(2) vec_from_1_to_2(1)]/norm(vec_from_1_to_2);
end

